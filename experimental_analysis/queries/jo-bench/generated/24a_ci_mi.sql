SELECT * FROM movie_info AS mi, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id;