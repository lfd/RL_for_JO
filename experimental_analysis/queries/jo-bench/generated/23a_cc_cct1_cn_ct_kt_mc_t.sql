SELECT * FROM company_name AS cn, kind_type AS kt, comp_cast_type AS cct1, title AS t, movie_companies AS mc, complete_cast AS cc, company_type AS ct WHERE cct1.kind = 'complete+verified' AND cn.country_code = '[us]' AND kt.kind IN ('movie') AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;