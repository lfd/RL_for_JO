SELECT * FROM cast_info AS ci, name AS n, movie_companies AS mc WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;