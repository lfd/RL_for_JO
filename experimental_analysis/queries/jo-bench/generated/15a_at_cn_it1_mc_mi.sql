SELECT * FROM info_type AS it1, movie_info AS mi, movie_companies AS mc, aka_title AS at, company_name AS cn WHERE cn.country_code = '[us]' AND it1.info = 'release dates' AND mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = at.movie_id AND at.movie_id = mi.movie_id AND mc.movie_id = at.movie_id AND at.movie_id = mc.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id;