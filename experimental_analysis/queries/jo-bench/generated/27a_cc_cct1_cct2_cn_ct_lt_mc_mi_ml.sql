SELECT * FROM comp_cast_type AS cct2, company_type AS ct, complete_cast AS cc, link_type AS lt, movie_link AS ml, movie_companies AS mc, movie_info AS mi, company_name AS cn, comp_cast_type AS cct1 WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND ct.kind = 'production companies' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mc.company_type_id = ct.id AND ct.id = mc.company_type_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;