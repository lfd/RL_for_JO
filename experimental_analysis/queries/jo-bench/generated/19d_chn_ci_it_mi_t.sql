SELECT * FROM title AS t, cast_info AS ci, char_name AS chn, movie_info AS mi, info_type AS it WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND t.production_year > 2000 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;