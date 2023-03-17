SELECT * FROM company_name AS cn, movie_companies AS mc, company_type AS ct, aka_title AS at, movie_info AS mi, title AS t WHERE cn.country_code = '[us]' AND mi.note LIKE '%internet%' AND t.production_year > 1990 AND t.id = at.movie_id AND at.movie_id = t.id AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;