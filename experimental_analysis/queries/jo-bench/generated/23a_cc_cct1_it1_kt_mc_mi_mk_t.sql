SELECT * FROM movie_companies AS mc, kind_type AS kt, title AS t, movie_info AS mi, info_type AS it1, movie_keyword AS mk, comp_cast_type AS cct1, complete_cast AS cc WHERE cct1.kind = 'complete+verified' AND it1.info = 'release dates' AND kt.kind IN ('movie') AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;