SELECT * FROM company_name AS cn, movie_companies AS mc, company_type AS ct, movie_keyword AS mk, movie_info AS mi WHERE cn.country_code = '[us]' AND mi.note LIKE '%internet%' AND mi.info LIKE 'USA:% 200%' AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id;