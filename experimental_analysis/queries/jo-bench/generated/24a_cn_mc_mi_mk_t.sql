SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t, movie_keyword AS mk, movie_info AS mi WHERE cn.country_code = '[us]' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND t.production_year > 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;