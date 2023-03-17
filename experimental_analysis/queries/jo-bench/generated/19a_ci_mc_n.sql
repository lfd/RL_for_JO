SELECT * FROM name AS n, cast_info AS ci, movie_companies AS mc WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%') AND n.gender = 'f' AND n.name LIKE '%Ang%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;