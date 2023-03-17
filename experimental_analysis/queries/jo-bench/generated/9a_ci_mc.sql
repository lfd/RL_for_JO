SELECT * FROM movie_companies AS mc, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;