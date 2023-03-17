SELECT * FROM char_name AS chn, cast_info AS ci, movie_info AS mi WHERE chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%') AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;