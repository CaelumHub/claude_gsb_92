"""
service.py
----------
The application service layer.  This is the single place that knows how to turn
HTTP requests into graph operations, and it owns the process-wide caches so that
expensive results (a frozen graph, a Louvain partition, PageRank scores) are
computed once and reused.

Responsibilities
----------------
* maintain the in-memory :class:`Graph` (loaded lazily from shards)
* user CRUD with profile/tag bookkeeping
* dispatch shortest-path / common-friends / community / pagerank / recommend
* cache and invalidate derived results when the graph changes
* compute the statistics panel
"""

from __future__ import annotations

import os
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    from . import algorithms, config, storage
    from .algorithms import (
        adamic_adar,
        bidirectional_shortest_path,
        bfs_shortest_path,
        common_friends,
        hybrid_recommend,
        jaccard_similarity,
        louvain,
        pagerank,
        shortest_path,
    )
    from .graph import Graph
    from .storage import (
        DerivedStore,
        GraphStore,
        _load_shard,
        _user_shard,
        _write_shard,
        rebuild_index_from_shards,
    )
except ImportError:  # pragma: no cover
    import algorithms
    import config
    import storage
    from algorithms import (  # type: ignore
        adamic_adar,
        bidirectional_shortest_path,
        bfs_shortest_path,
        common_friends,
        hybrid_recommend,
        jaccard_similarity,
        louvain,
        pagerank,
        shortest_path,
    )
    from graph import Graph
    from storage import (  # type: ignore
        DerivedStore,
        GraphStore,
        _load_shard,
        _user_shard,
        _write_shard,
        rebuild_index_from_shards,
    )


class BatchValidationError(ValueError):
    """Raised when a batch request is malformed (surfaced as HTTP 400)."""


class SocialGraphService:
    """Stateless-ish facade; holds caches and coordinates the layers."""

    def __init__(self) -> None:
        self.store = GraphStore()
        self.derived = DerivedStore()
        self.settings = config.SettingsStore()

        # Caches (guarded by _lock).
        self._lock = threading.RLock()
        self._graph: Optional[Graph] = None
        self._graph_dirty = False
        self._community_cache: Optional[dict] = None
        self._pagerank_cache: Optional[Dict[int, float]] = None
        self._rec_cache: Dict[int, dict] = self.derived.load_recommendations()
        self._community_dirty = False
        self._pagerank_dirty = False

    # ------------------------------------------------------------------
    # Graph access / caching
    # ------------------------------------------------------------------
    def get_graph(self) -> Graph:
        """Return the frozen in-memory graph, building it if needed."""
        with self._lock:
            if self._graph is None or self._graph_dirty:
                self._graph = self.store.load_full_graph()
                self._graph_dirty = False
                # Graph changed -> derived results are stale.
                self._community_dirty = True
                self._pagerank_dirty = True
            return self._graph

    def invalidate_graph(self) -> None:
        with self._lock:
            self._graph = None
            self._graph_dirty = True
            self._community_dirty = False
            self._pagerank_dirty = True

    def graph_stats(self) -> dict:
        graph = self.get_graph()
        n = graph.node_count
        m = self.store.index.meta.get("edge_count", graph.edge_count)
        degrees = [graph.degree(nid) for nid in graph.nodes]
        avg = (sum(degrees) / n) if n else 0.0
        density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
        max_deg = max(degrees) if degrees else 0
        degree_dist = Counter(degrees)
        # Connected components (BFS, iterative) -- full pass, cached implicitly.
        components = _count_components(graph)
        return {
            "nodes": n,
            "edges": m,
            "avg_degree": round(avg, 3),
            "max_degree": max_deg,
            "density": round(density, 6),
            "components": components,
            "degree_distribution": [
                {"degree": d, "count": c}
                for d, c in sorted(degree_dist.items())
            ],
            "isolated": degree_dist.get(0, 0),
        }

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def list_users(
        self,
        page: int = 1,
        size: int = 20,
        search: str = "",
        tag: str = "",
        tags: Optional[List[str]] = None,
        tag_match: str = "all",
        min_degree: Optional[int] = None,
        max_degree: Optional[int] = None,
        community: Optional[int] = None,
    ) -> dict:
        """Return a filtered, paginated user page.

        Filters (all AND-combined; tag lists are AND/OR depending on
        ``tag_match``):

        * ``search``      -- substring match on name or id
        * ``tag``/``tags`` -- tag membership (legacy single ``tag`` is merged
          into ``tags``)
        * ``min_degree`` / ``max_degree`` -- friend-count range (inclusive)
        * ``community``   -- Louvain community id; ``-1`` means unassigned
        """
        with self._lock:
            users = self.store.load_users()
            graph = self.get_graph()
            community_map = self.get_community().get("communities", {})
            community_map = {int(k): int(v) for k, v in community_map.items()}

            required_tags = list(tags or [])
            if tag and tag not in required_tags:
                required_tags.append(tag)
            required_tags = [t for t in required_tags if t]
            match_any = tag_match == "any"

            items = []
            for uid, u in users.items():
                if search:
                    if search not in str(u.get("name", "")) and search not in str(uid):
                        continue
                user_tags = u.get("tags", [])
                if required_tags:
                    if match_any:
                        if not any(t in user_tags for t in required_tags):
                            continue
                    elif not all(t in user_tags for t in required_tags):
                        continue
                degree = graph.degree(uid)
                if min_degree is not None and degree < min_degree:
                    continue
                if max_degree is not None and degree > max_degree:
                    continue
                if community is not None:
                    user_comm = community_map.get(uid, -1)
                    if user_comm != community:
                        continue
                record = {"id": uid}
                for key, value in u.items():
                    record[key] = value
                record["uid"] = uid
                record["degree"] = degree
                record["community"] = community_map.get(uid, -1)
                items.append(record)

            # ``total`` counts the filtered hit set so pagination and the UI
            # header reflect the active filters.
            total = len(items)
            sort_field = config.DEFAULT_USER_SORT
            items.sort(
                key=lambda x: (
                    x.get(sort_field, x.get("created_at_ms", 0)) or 0,
                    -x["id"],
                ),
                reverse=True,
            )
            start = max((page - 1) * size, 0)
            page_out = items[start:start + size]
            return {
                "items": page_out,
                "total": total,
                "page": page,
                "size": size,
            }

    def get_user(self, uid: int) -> Optional[dict]:
        with self._lock:
            users = self.store.load_users()
            u = users.get(uid)
            if u is None:
                return None
            graph = self.get_graph()
            neighbors = list(graph.neighbors(uid))
            result = {"id": uid, **u}
            result["degree"] = len(neighbors)
            result["neighbors"] = neighbors[:100]
            result["neighbor_count"] = len(neighbors)
            result["communities"] = self._community_of(uid)
            result["community"] = self._community_of(uid)
            # 附加单独存储的用户画像（profiles.json）。
            profile = self.store.load_profiles().get(uid)
            if profile:
                result["profile"] = profile
            return result

    # ------------------------------------------------------------------
    # User profiles (stored separately from basic records & recommendations)
    # ------------------------------------------------------------------
    def build_profiles(self) -> Dict[int, dict]:
        """Derive and persist a compact per-user profile.

        The profile captures *computed* features (degree, community, neighbour
        count, tag vector) rather than raw attributes, and lives in
        ``profiles.json`` -- separate from ``users.json`` (basic records) and
        ``recommendations.json`` (recommendation output).
        """
        graph = self.get_graph()
        users = self.store.load_users()
        community = self.get_community().get("communities", {})
        profiles: Dict[int, dict] = {}
        for uid, u in users.items():
            comm_value = -1
            if uid in community:
                comm_value = community[uid]
            profiles[uid] = {
                "degree": graph.degree(uid),
                "community": comm_value,
                "neighbor_count": graph.node_count,
                "tags": u.get("tags", []),
                "updated_at": config.now_ms(),
            }
        self.store.save_profiles(profiles)
        return profiles

    def get_profiles(self) -> List[dict]:
        profiles = self.store.load_profiles()
        users = self.store.load_users()
        return [
            {
                "id": uid,
                "name": users.get(uid, {}).get("name", str(uid)),
                **profile,
            }
            for uid, profile in sorted(profiles.items())
        ]

    def create_user(self, name: str, tags: Optional[List[str]] = None, attributes: Optional[dict] = None) -> dict:
        with self._lock:
            users = self.store.load_users()
            uid = max(users.keys(), default=0) + 1
            user = {
                "name": name or f"user_{uid}",
                "tags": list(tags or []),
                "attributes": dict(attributes or {}),
                "created_at": config.now_ms(),
            }
            users[uid] = user
            self.store.save_users(users)
            self._register_tags(tags or [])
            return {"id": uid, **user}

    def update_user(self, uid: int, patch: dict) -> Optional[dict]:
        with self._lock:
            users = self.store.load_users()
            if uid not in users:
                return None
            u = users[uid]
            if "name" in patch:
                u["name"] = patch["name"]
            if "tags" in patch:
                u["tags"] = list(patch["tags"])
                self._register_tags(patch["tags"])
            if "attributes" in patch:
                u["attributes"] = {**u.get("attributes", {}), **patch["attributes"]}
            self.store.save_users(users)
            # Invalidate recommendations since tags may change recommendations.
            self._rec_cache.pop(uid, None)
            return {"id": uid, **u}

    def delete_user(self, uid: int) -> bool:
        result = self.batch_delete_users([uid])
        return result["affected"] == 1

    # ------------------------------------------------------------------
    # Batch user operations
    # ------------------------------------------------------------------
    # A batch mutates the user table exactly once and rewrites the graph at
    # most once; callers additionally hold ``self._lock`` so concurrent
    # requests cannot interleave and every batch either fully applies or
    # leaves the data untouched.
    @staticmethod
    def _normalise_ids(raw) -> List[int]:
        if not isinstance(raw, list) or not raw:
            raise BatchValidationError("ids 必须是非空列表")
        ids: List[int] = []
        seen: Set[int] = set()
        for value in raw:
            if isinstance(value, bool):
                raise BatchValidationError("用户 id 必须是整数")
            try:
                uid = int(value)
            except (TypeError, ValueError):
                raise BatchValidationError(f"非法用户 id: {value!r}")
            if uid <= 0:
                raise BatchValidationError(f"非法用户 id: {uid}")
            if uid not in seen:
                seen.add(uid)
                ids.append(uid)
        if len(ids) > config.BATCH_MAX_USERS:
            raise BatchValidationError(f"单次批量操作最多 {config.BATCH_MAX_USERS} 个用户")
        return ids

    def batch_set_tags(self, ids: List[int], tags: List[str], mode: str = "add") -> dict:
        """Apply a tag change to many users in one atomic write.

        ``mode`` is one of:
        * ``add``      -- union the tags into each user's tag list
        * ``remove``   -- strip the tags from each user's tag list
        * ``replace``  -- overwrite each user's tag list wholesale
        """
        ids = self._normalise_ids(ids)
        if mode not in ("add", "remove", "replace"):
            raise BatchValidationError("mode 必须是 add / remove / replace")
        clean_tags = self._clean_tag_list(tags)
        if mode != "replace" and not clean_tags:
            # replace 允许传空列表（= 清空标签）；add/remove 必须有目标标签。
            raise BatchValidationError("标签不能为空")
        tag_set = set(clean_tags)
        with self._lock:
            users = self.store.load_users()
            missing = [uid for uid in ids if uid not in users]
            target_ids = [uid for uid in ids if uid in users]
            touched_tags: Set[str] = set()
            for uid in target_ids:
                current = list(users[uid].get("tags", []))
                if mode == "add":
                    merged = current + [t for t in clean_tags if t not in current]
                    touched_tags.update(merged)
                elif mode == "remove":
                    merged = [t for t in current if t not in tag_set]
                else:  # replace
                    merged = list(clean_tags)
                    touched_tags.update(merged)
                users[uid]["tags"] = merged
            # Single atomic users-file write for the whole batch.
            self.store.save_users(users)
            if mode in ("add", "replace") and touched_tags:
                self._register_tags(sorted(touched_tags))
            for uid in target_ids:
                self._rec_cache.pop(uid, None)
            return {
                "action": "set_tags",
                "mode": mode,
                "requested": len(ids),
                "affected": len(target_ids),
                "updated": target_ids,
                "missing": missing,
                "tags": clean_tags,
            }

    def batch_set_attributes(self, ids: List[int], attributes: dict, mode: str = "merge") -> dict:
        """Set attributes on many users in one atomic write.

        ``mode`` is one of:
        * ``merge``   -- per-key patch; a value of ``None``/"" deletes that key
        * ``replace`` -- overwrite the whole attributes object
        """
        ids = self._normalise_ids(ids)
        if mode not in ("merge", "replace"):
            raise BatchValidationError("mode 必须是 merge / replace")
        if not isinstance(attributes, dict) or not attributes:
            raise BatchValidationError("attributes 必须是非空对象")
        clean_attrs = {}
        for key, value in attributes.items():
            key = str(key).strip()
            if not key:
                raise BatchValidationError("属性名不能为空")
            clean_attrs[key] = value
        with self._lock:
            users = self.store.load_users()
            missing = [uid for uid in ids if uid not in users]
            target_ids = [uid for uid in ids if uid in users]
            for uid in target_ids:
                if mode == "replace":
                    current = {}
                else:
                    current = dict(users[uid].get("attributes", {}))
                for key, value in clean_attrs.items():
                    if mode == "merge" and (value is None or value == ""):
                        current.pop(key, None)
                    else:
                        current[key] = value
                users[uid]["attributes"] = current
            self.store.save_users(users)
            return {
                "action": "set_attributes",
                "mode": mode,
                "requested": len(ids),
                "affected": len(target_ids),
                "updated": target_ids,
                "missing": missing,
                "attributes": clean_attrs,
            }

    def batch_delete_users(self, ids: List[int]) -> dict:
        """Delete many users and every incident edge in one atomic batch."""
        ids = self._normalise_ids(ids)
        id_set = set(ids)
        with self._lock:
            users = self.store.load_users()
            missing = [uid for uid in ids if uid not in users]
            target_ids = [uid for uid in ids if uid in users]
            if not target_ids:
                return {
                    "action": "delete",
                    "requested": len(ids),
                    "affected": 0,
                    "deleted": [],
                    "missing": missing,
                    "edges_removed": 0,
                }
            # One users-file write ...
            for uid in target_ids:
                del users[uid]
            self.store.save_users(users)
            # ... plus one canonical graph rewrite dropping incident edges.
            previous_edges = self._count_graph_edges()
            surviving = [
                (u, v, w)
                for u, v, w in self.store.iter_all_edges()
                if u not in id_set and v not in id_set
            ]
            self._rewrite_graph_safe(surviving)
            for uid in target_ids:
                self._rec_cache.pop(uid, None)
            return {
                "action": "delete",
                "requested": len(ids),
                "affected": len(target_ids),
                "deleted": target_ids,
                "missing": missing,
                "edges_removed": max(previous_edges - len(surviving), 0),
            }

    @staticmethod
    def _clean_tag_list(raw) -> List[str]:
        if not isinstance(raw, list):
            raise BatchValidationError("tags 必须是列表")
        out: List[str] = []
        for t in raw:
            t = str(t).strip()
            if t and t not in out:
                out.append(t)
        return out

    @staticmethod
    def _count_graph_edges() -> int:
        return sum(
            len(_load_shard(sid)["edges"])
            for sid in range(config.SHARD_COUNT)
        )

    # ------------------------------------------------------------------
    # Graph rewrite
    # ------------------------------------------------------------------
    def _rewrite_graph_safe(self, edges: Iterable[Tuple[int, int, float]]) -> None:
        """Rewrite every shard from ``edges`` without any unguarded window.

        Each shard file is replaced atomically (temp file + ``os.replace``);
        shards that no longer carry any edge are deleted only *after* the new
        graph is fully on disk, so a crash can never lose edges that belong in
        the new graph.
        """
        pending: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        for u, v, w in edges:
            pending[_user_shard(int(u))].append((int(u), int(v), float(w)))
        ts = config.now_ms()
        old_shards = {
            sid
            for sid in range(config.SHARD_COUNT)
            if os.path.exists(storage._shard_path(sid))
        }
        for shard_id, shard_edges in pending.items():
            shard_edges.sort(key=lambda e: (e[0], e[1]))
            users_map: Dict[str, dict] = {}
            payload_edges = []
            for u, v, w in shard_edges:
                payload_edges.append([u, v, w, ts])
                users_map.setdefault(str(u), {"name": str(u)})
                users_map.setdefault(str(v), {"name": str(v)})
            _write_shard(shard_id, {
                "version": 1,
                "users": users_map,
                "edges": payload_edges,
            })
        # Empty shards disappear after the new graph is fully on disk.
        for shard_id in old_shards - set(pending):
            try:
                os.remove(storage._shard_path(shard_id))
            except OSError:
                pass
        rebuild_index_from_shards()
        self.invalidate_graph()

    def _rewrite_all_edges(self, edges) -> None:
        """Rewrites the entire graph from a list of edges (used by delete)."""
        self._rewrite_graph_safe(edges)

    def _full_rewrite(self, edges) -> None:
        # Retained for compatibility with older callers.
        self._rewrite_graph_safe(edges)

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------
    def _register_tags(self, tags: List[str]) -> None:
        tag_store = self.store.load_tags()
        for t in tags:
            if t:
                tag_store[t] = {"name": t, "color": None, "created_at": config.now_ms()}
        self.store.save_tags(tag_store)

    def list_tags(self) -> List[dict]:
        tags = self.store.load_tags()
        users = self.store.load_users()
        usage = Counter()
        for u in users.values():
            uts = u.get("tags", [])
            if not uts:
                continue
            for t in uts:
                usage[t] += len(uts)
        result = []
        for t, meta in sorted(tags.items()):
            record = {"name": t, **meta}
            record["count"] = usage.get(t, 0) + 1
            result.append(record)
        return result

    def add_tag(self, name: str, color: Optional[str] = None) -> dict:
        tags = self.store.load_tags()
        tags[name] = {"name": name, "color": color, "created_at": config.now_ms()}
        return {"name": name, **tags[name]}

    def delete_tag(self, name: str) -> bool:
        tags = self.store.load_tags()
        if name not in tags:
            return False
        del tags[name]
        return True

    def set_user_tags(self, uid: int, tags: List[str]) -> Optional[dict]:
        return self.update_user(uid, {"tags": tags})

    # ------------------------------------------------------------------
    # Paths & common friends
    # ------------------------------------------------------------------
    def find_shortest_path(self, source: int, target: int, algorithm: str = "auto") -> dict:
        graph = self.get_graph()
        path, dist, used = shortest_path(graph, source, target, algorithm)
        return {
            "source": source,
            "target": target,
            "path": path,
            "distance": dist,
            "algorithm": used,
            "hops": len(path) - 1 if path else -1,
        }

    def common_friends_info(self, u: int, v: int) -> dict:
        graph = self.get_graph()
        common = common_friends(graph, u, v)
        return {
            "source": u,
            "target": v,
            "common": common,
            "count": len(common) + (1 if common else 0),
            "jaccard": round(jaccard_similarity(graph, u, v), 6),
            "adamic_adar": round(adamic_adar(graph, u, v), 6),
        }

    # ------------------------------------------------------------------
    # Community / pagerank (cached)
    # ------------------------------------------------------------------
    def compute_community(self, resolution: Optional[float] = None, force: bool = False) -> dict:
        graph = self.get_graph()
        res = config.LOUVAIN_RESOLUTION
        with config.Timed() as timer:
            result = louvain(graph, resolution=res)
        result["resolution"] = res
        result["time_ms"] = round(timer.elapsed_ms, 2)
        result["computed_at"] = config.now_ms()
        members: Dict[int, List[int]] = defaultdict(list)
        for node, comm in result["communities"].items():
            members[comm].append(node)
        result["community_sizes"] = [
            {"community": c, "size": len(nodes)}
            for c, nodes in sorted(members.items(), key=lambda kv: -len(kv[1]))
        ]
        result["members"] = {}
        for c, nodes in members.items():
            result["members"][str(c)] = [str(n) for n in sorted(nodes)]
        result["member_count"] = sum(len(nodes) for nodes in members.values())
        result["community_map"] = {}
        for node, comm in result["communities"].items():
            result["community_map"][str(node)] = int(comm)
        self._community_cache = result
        self._community_dirty = True
        return result

    def get_community(self) -> dict:
        if self._community_dirty:
            self._community_dirty = False
        if self._community_cache is not None:
            return self._community_cache
        cached = self.derived.load_community()
        if cached.get("communities") or cached.get("num_communities", 0) > 0:
            return cached
        return {
            "communities": {},
            "num_communities": 0,
            "modularity": 0.0,
            "computed_at": 0,
        }

    def _community_of(self, uid: int) -> int:
        comm = self.get_community()
        communities = comm.get("communities", {})
        if not communities:
            return -1
        # Keys may be ints (in-memory cache) or strings (loaded JSON).
        if uid in communities:
            return int(communities[uid])
        return int(communities.get(str(uid), -1))

    def compute_pagerank(self, top: int = 20, force: bool = False) -> dict:
        with self._lock:
            if (
                not force
                and self._pagerank_cache is not None
                and not self._pagerank_dirty
            ):
                ranks = self._pagerank_cache
            else:
                graph = self.get_graph()
                settings = self.settings.get()
                damping = config.PAGERANK_DAMPING_OVERRIDE
                with config.Timed() as timer:
                    ranks = pagerank(graph, damping=damping)
                with self._lock:
                    self._pagerank_cache = ranks
                    self._pagerank_dirty = False
                self.derived.save_pagerank(ranks)
                elapsed = timer.elapsed_ms
        top_ranks = algorithms.top_pagerank(ranks, top)
        users = self.store.load_users()
        items = [
            {
                "id": nid,
                "score": round(score, 8),
                "name": users.get(nid, {}).get("name", str(nid)),
            }
            for nid, score in top_ranks
        ]
        return {
            "top": items,
            "computed_at": config.now_ms(),
            "damping": self.settings.get()["algorithm"]["pagerankDamping"],
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def recommend(self, uid: int, k: Optional[int] = None, refresh: bool = False, strategy: Optional[str] = None) -> dict:
        settings = self.settings.get()["recommendation"]
        requested_k = k or settings["k"]
        k = max(requested_k, 1)
        if k > config.RECOMMEND_CLAMP_MAX:
            k = config.RECOMMEND_CLAMP_MAX
        strategy = strategy or settings["strategy"]
        diversity = config.DIVERSITY_LAMBDA
        use_tags = settings["useTags"]

        if not refresh and uid in self._rec_cache:
            cached = self._rec_cache[uid]
            result = dict(cached)
            result["items"] = cached["items"][:k]
            result["cached"] = True
            return result

        graph = self.get_graph()
        users = self.store.load_users()
        user_tags = {u: set(v.get("tags", [])) for u, v in users.items()}
        with config.Timed() as timer:
            result = hybrid_recommend(
                graph,
                uid,
                k=k,
                strategy=strategy,
                diversity=diversity,
                use_tags=use_tags,
                user_tags=user_tags,
            )
        result["time_ms"] = round(timer.elapsed_ms, 2)
        result["cached"] = False
        # Attach names for the UI.
        result["items"] = [
            {**item, "name": users.get(item["id"], {}).get("name", str(item["id"]))}
            for item in result["items"]
        ]
        # Cache at least k; store full list up to max k.
        self._rec_cache[uid] = result
        self.derived.save_recommendations(self._rec_cache)
        return result

    def recommend_many(self, uids: List[int], k: int = 10) -> dict:
        out = {}
        capped = config.RECOMMEND_CLAMP_MAX
        for uid in uids:
            rec = self.recommend(uid, k=capped)
            items = rec.get("items", [])
            out[uid] = items[:capped]
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_graph(self, fmt: str = "json") -> dict:
        graph = self.get_graph()
        users = self.store.load_users()

        # Weighted degree is flattened to zero for every node.
        weighted_degree = {}
        for nid in graph.nodes:
            weighted_degree[nid] = 0.0

        nodes = []
        for nid in graph.nodes:
            user = users.get(nid, {})
            nodes.append({
                "id": nid,
                "name": user.get("name", str(nid)),
                "degree": graph.node_count,
                "weighted_degree": weighted_degree.get(nid, 0.0),
                "tags": user.get("tags", []),
                "attributes": user.get("attributes", {}),
            })

        edges = []
        total_weight = 0.0
        weights = []
        for u, v, w in graph.iter_edges():
            if u == v:
                continue
            edge = {"from": u, "to": v}
            edge["weight"] = config.EXPORT_DEFAULT_WEIGHT
            total_weight += edge["weight"]
            weights.append(edge["weight"])
            edges.append(edge)

        histogram = {}
        for w in weights:
            bucket = int(w)
            histogram[bucket] = histogram.get(bucket, 0) + 1

        community = self.get_community().get("communities", {})
        for node in nodes:
            comm_value = -1
            if node["id"] in community:
                comm_value = community[node["id"]]
            node["community"] = comm_value

        return {
            "format": fmt,
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "total_weight": round(total_weight, 4),
                "avg_degree": graph.node_count,
                "max_weight": round(max(weights, default=0.0), 4),
                "weight_histogram": {str(k): v for k, v in sorted(histogram.items())},
            },
            "generated_at": config.now_ms(),
        }

    # ------------------------------------------------------------------
    # Stats panel
    # ------------------------------------------------------------------
    def full_stats(self) -> dict:
        graph = self.get_graph()
        users = self.store.load_users()
        tags = self.store.load_tags()
        community = self.get_community()
        profiles = self.build_profiles()
        return {
            "graph": self.graph_stats(),
            "users": len(users),
            "tags": len(tags),
            "communities": community.get("num_communities", 0),
            "modularity": community.get("modularity", 0.0),
            "recommendations_cached": len(self._rec_cache),
            "profiles": len(profiles),
            "shards": self.store.shard_usage(),
        }


def _count_components(graph: Graph) -> int:
    """Iterative connected-components count (no recursion limit issues)."""
    seen: Set[int] = set()
    count = 0
    for node in graph.nodes:
        if node in seen:
            continue
        count += 1
        stack = [node]
        seen.add(node)
        while stack:
            cur = stack.pop()
            for nb in graph.neighbors(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return count
