SELECT * FROM movie_companies AS mc, info_type AS it, movie_info AS mi, comp_cast_type AS cct1, complete_cast AS cc, comp_cast_type AS cct2, role_type AS rt, name AS n, title AS t, keyword AS k, movie_keyword AS mk, cast_info AS ci, char_name AS chn, person_info AS pi, info_type AS it3 WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND it3.info = 'height' AND k.keyword = 'computer-animation' AND mi.info LIKE 'USA:%200%' AND n.gender = 'f' AND n.name LIKE '%An%' AND rt.role = 'actress' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND n.id = pi.person_id AND pi.person_id = n.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND it3.id = pi.info_type_id AND pi.info_type_id = it3.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;