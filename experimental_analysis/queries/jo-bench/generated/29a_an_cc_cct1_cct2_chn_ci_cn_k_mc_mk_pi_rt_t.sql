SELECT * FROM keyword AS k, person_info AS pi, role_type AS rt, title AS t, movie_companies AS mc, company_name AS cn, char_name AS chn, cast_info AS ci, movie_keyword AS mk, aka_name AS an, comp_cast_type AS cct2, comp_cast_type AS cct1, complete_cast AS cc WHERE cct1.kind = 'cast' AND cct2.kind = 'complete+verified' AND chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND k.keyword = 'computer-animation' AND rt.role = 'actress' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND ci.person_id = pi.person_id AND pi.person_id = ci.person_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;