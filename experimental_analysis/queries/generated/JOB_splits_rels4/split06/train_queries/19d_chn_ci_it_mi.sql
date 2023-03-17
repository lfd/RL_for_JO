SELECT * FROM info_type AS it, cast_info AS ci, movie_info AS mi, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND it.info = 'release dates' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;