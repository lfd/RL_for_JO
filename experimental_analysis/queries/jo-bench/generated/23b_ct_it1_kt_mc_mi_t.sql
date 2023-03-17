SELECT * FROM movie_companies AS mc, info_type AS it1, kind_type AS kt, title AS t, movie_info AS mi, company_type AS ct WHERE it1.info = 'release dates' AND kt.kind IN ('movie') AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;