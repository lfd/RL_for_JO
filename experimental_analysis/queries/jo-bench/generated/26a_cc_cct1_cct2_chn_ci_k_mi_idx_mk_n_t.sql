SELECT * FROM movie_keyword AS mk, comp_cast_type AS cct2, complete_cast AS cc, title AS t, keyword AS k, cast_info AS ci, movie_info_idx AS mi_idx, char_name AS chn, name AS n, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence', 'magnet', 'web', 'claw', 'laser') AND mi_idx.info > '7.0' AND t.production_year > 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND cc.movie_id = mi_idx.movie_id AND mi_idx.movie_id = cc.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = ci.person_id AND ci.person_id = n.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;