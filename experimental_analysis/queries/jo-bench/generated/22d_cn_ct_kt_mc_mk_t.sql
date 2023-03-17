SELECT * FROM movie_keyword AS mk, title AS t, movie_companies AS mc, company_name AS cn, company_type AS ct, kind_type AS kt WHERE cn.country_code != '[us]' AND kt.kind IN ('movie', 'episode') AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cn.id = mc.company_id AND mc.company_id = cn.id;