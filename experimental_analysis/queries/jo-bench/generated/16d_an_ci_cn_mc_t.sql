SELECT * FROM title AS t, company_name AS cn, movie_companies AS mc, cast_info AS ci, aka_name AS an WHERE cn.country_code = '[us]' AND t.episode_nr >= 5 AND t.episode_nr < 100 AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_id = cn.id AND cn.id = mc.company_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;