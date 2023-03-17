SELECT * FROM info_type AS it, movie_info AS mi, title AS t, comp_cast_type AS cct1, complete_cast AS cc, movie_companies AS mc, company_name AS cn, aka_name AS an, keyword AS k, role_type AS rt, cast_info AS ci, movie_keyword AS mk WHERE cct1.kind = 'cast' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND it.info = 'release dates' AND k.keyword = 'computer-animation' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND rt.role = 'actress' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;