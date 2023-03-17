SELECT * FROM movie_companies AS mc, company_name AS cn, complete_cast AS cc, comp_cast_type AS cct2, comp_cast_type AS cct1, movie_info AS mi WHERE cct1.kind = 'crew' AND cct2.kind != 'complete+verified' AND cn.country_code != '[us]' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;