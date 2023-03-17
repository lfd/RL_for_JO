SELECT * FROM company_name AS cn, keyword AS k, movie_keyword AS mk, complete_cast AS cc, comp_cast_type AS cct1, movie_link AS ml, link_type AS lt, movie_companies AS mc WHERE cct1.kind IN ('cast', 'crew') AND cn.country_code != '[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND k.keyword = 'sequel' AND lt.link LIKE '%follow%' AND mc.note IS NULL AND lt.id = ml.link_type_id AND ml.link_type_id = lt.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mc.company_id = cn.id AND cn.id = mc.company_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = cc.movie_id AND cc.movie_id = ml.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;