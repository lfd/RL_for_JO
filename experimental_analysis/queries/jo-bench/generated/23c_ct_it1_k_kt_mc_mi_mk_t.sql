SELECT * FROM movie_companies AS mc, company_type AS ct, movie_info AS mi, title AS t, movie_keyword AS mk, kind_type AS kt, info_type AS it1, keyword AS k WHERE it1.info = 'release dates' AND kt.kind IN ('movie', 'tv movie', 'video movie', 'video game') AND mi.note LIKE '%internet%' AND mi.info IS NOT NULL AND (mi.info LIKE 'USA:% 199%' OR mi.info LIKE 'USA:% 200%') AND t.production_year > 1990 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;