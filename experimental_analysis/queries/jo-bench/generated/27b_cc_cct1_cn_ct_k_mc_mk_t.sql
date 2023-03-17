SELECT * FROM keyword AS k, company_type AS ct, company_name AS cn, movie_companies AS mc, complete_cast AS cc, movie_keyword AS mk, comp_cast_type AS cct1, title AS t WHERE cct1.kind IN ('cast', 'crew') AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind = 'production companies' AND k.keyword = 'sequel' AND mc.note IS NULL AND t.production_year = 1998 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;