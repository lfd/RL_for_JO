SELECT * FROM complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, title AS t, movie_companies AS mc, company_type AS ct, company_name AS cn WHERE cct1.kind = 'cast' AND cct2.kind LIKE 'complete%' AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind = 'production companies' AND mc.note IS NULL AND t.production_year BETWEEN 1950 AND 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;