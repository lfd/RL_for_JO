SELECT * FROM movie_info AS mi, complete_cast AS cc, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND mi.info LIKE 'USA:%200%' AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id;