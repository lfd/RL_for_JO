SELECT * FROM title AS t, kind_type AS kt, movie_companies AS mc, company_name AS cn, company_type AS ct WHERE cn.country_code = '[us]' AND kt.kind IN ('movie', 'tv movie', 'video movie', 'video game') AND t.production_year > 1990 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;