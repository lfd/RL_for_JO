SELECT * FROM keyword AS k, movie_keyword AS mk, cast_info AS ci, info_type AS it2, movie_info_idx AS mi_idx, title AS t, kind_type AS kt, char_name AS chn WHERE chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND it2.info = 'rating' AND k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight', 'violence', 'magnet', 'web', 'claw', 'laser') AND kt.kind = 'movie' AND mi_idx.info > '7.0' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mk.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it2.id = mi_idx.info_type_id AND mi_idx.info_type_id = it2.id;