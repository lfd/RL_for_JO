SELECT * FROM char_name AS chn, cast_info AS ci, aka_name AS an, role_type AS rt, movie_keyword AS mk, movie_info AS mi, complete_cast AS cc, info_type AS it, title AS t, movie_companies AS mc, company_name AS cn, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND it.info = 'release dates' AND mi.info LIKE 'USA:%200%' AND rt.role = 'actress' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;