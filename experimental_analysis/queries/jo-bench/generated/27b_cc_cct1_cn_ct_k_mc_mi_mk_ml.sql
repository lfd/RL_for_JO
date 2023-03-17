SELECT * FROM company_type AS ct, movie_companies AS mc, movie_info AS mi, company_name AS cn, movie_keyword AS mk, movie_link AS ml, complete_cast AS cc, comp_cast_type AS cct1, keyword AS k WHERE cct1.kind IN ('cast', 'crew') AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind = 'production companies' AND k.keyword = 'sequel' AND mc.note IS NULL AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;