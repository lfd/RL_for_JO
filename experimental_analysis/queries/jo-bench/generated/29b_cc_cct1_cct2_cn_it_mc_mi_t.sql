SELECT * FROM title AS t, movie_companies AS mc, company_name AS cn, comp_cast_type AS cct1, complete_cast AS cc, info_type AS it, movie_info AS mi, comp_cast_type AS cct2 WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND cn.country_code = '[us]' AND it.info = 'release dates' AND mi.info LIKE 'USA:%200%' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;