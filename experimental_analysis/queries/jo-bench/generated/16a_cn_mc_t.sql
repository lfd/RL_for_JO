SELECT * FROM company_name AS cn, movie_companies AS mc, title AS t WHERE cn.country_code = '[us]' AND t.episode_nr >= 50 AND t.episode_nr < 100 AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id;