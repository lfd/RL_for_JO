SELECT * FROM name AS n, title AS t, cast_info AS ci, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;