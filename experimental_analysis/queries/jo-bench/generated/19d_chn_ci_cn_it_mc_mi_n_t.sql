SELECT * FROM title AS t, company_name AS cn, movie_companies AS mc, cast_info AS ci, char_name AS chn, movie_info AS mi, info_type AS it, name AS n WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND it.info = 'release dates' AND n.gender = 'f' AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;