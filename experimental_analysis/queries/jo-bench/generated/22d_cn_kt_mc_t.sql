SELECT * FROM kind_type AS kt, company_name AS cn, movie_companies AS mc, title AS t WHERE cn.country_code != '[us]' AND kt.kind IN ('movie', 'episode') AND t.production_year > 2005 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND cn.id = mc.company_id AND mc.company_id = cn.id;