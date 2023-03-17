SELECT * FROM movie_companies AS mc, company_name AS cn, movie_info AS mi, title AS t, movie_keyword AS mk WHERE cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND t.production_year > 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;