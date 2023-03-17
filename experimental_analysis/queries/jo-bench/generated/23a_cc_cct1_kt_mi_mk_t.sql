SELECT * FROM movie_keyword AS mk, comp_cast_type AS cct1, kind_type AS kt, title AS t, complete_cast AS cc, movie_info AS mi WHERE cct1.kind = 'complete+verified' AND kt.kind IN ('movie') AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;