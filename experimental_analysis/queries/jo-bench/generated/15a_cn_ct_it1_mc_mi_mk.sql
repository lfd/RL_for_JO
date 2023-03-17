SELECT * FROM company_type AS ct, company_name AS cn, movie_companies AS mc, movie_keyword AS mk, info_type AS it1, movie_info AS mi WHERE cn.country_code = '[us]' AND it1.info = 'release dates' AND mc.note LIKE '%(200%)%' AND mc.note LIKE '%(worldwide)%' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;