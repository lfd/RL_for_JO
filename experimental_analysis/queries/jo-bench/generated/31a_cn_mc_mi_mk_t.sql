SELECT * FROM movie_info AS mi, movie_companies AS mc, company_name AS cn, title AS t, movie_keyword AS mk WHERE cn.name LIKE 'Lionsgate%' AND mi.info IN ('Horror', 'Thriller') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;