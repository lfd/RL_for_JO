SELECT * FROM name AS n, title AS t, char_name AS chn, comp_cast_type AS cct1, complete_cast AS cc, cast_info AS ci, movie_info AS mi, info_type AS it WHERE cct1.kind = 'cast' AND chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND mi.info LIKE 'USA:%200%' AND n.gender = 'f' AND n.name LIKE '%An%' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id;