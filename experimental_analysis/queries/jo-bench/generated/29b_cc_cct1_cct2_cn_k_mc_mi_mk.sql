SELECT * FROM keyword AS k, movie_keyword AS mk, movie_info AS mi, movie_companies AS mc, company_name AS cn, comp_cast_type AS cct2, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND cn.country_code = '[us]' AND k.keyword = 'computer-animation' AND mi.info LIKE 'USA:%200%' AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;