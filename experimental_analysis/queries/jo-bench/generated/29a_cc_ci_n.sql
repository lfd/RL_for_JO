SELECT * FROM name AS n, complete_cast AS cc, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id;