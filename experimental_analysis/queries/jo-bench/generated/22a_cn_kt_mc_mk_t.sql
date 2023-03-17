SELECT * FROM movie_keyword AS mk, movie_companies AS mc, title AS t, company_name AS cn, kind_type AS kt WHERE cn.country_code != '[us]' AND kt.kind IN ('movie', 'episode') AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2008 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;