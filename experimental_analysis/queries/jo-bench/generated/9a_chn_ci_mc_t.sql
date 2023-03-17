SELECT * FROM cast_info AS ci, char_name AS chn, movie_companies AS mc, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND t.production_year BETWEEN 2005 AND 2015 AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;