SELECT * FROM company_name AS cn, movie_companies AS mc, aka_title AS at, movie_info AS mi WHERE cn.country_code = '[us]' AND mi.note LIKE '%internet%' AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;