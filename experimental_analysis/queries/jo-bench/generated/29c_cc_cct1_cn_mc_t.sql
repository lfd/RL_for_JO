SELECT * FROM title AS t, comp_cast_type AS cct1, complete_cast AS cc, movie_companies AS mc, company_name AS cn WHERE cct1.kind = 'cast' AND cn.country_code = '[us]' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;