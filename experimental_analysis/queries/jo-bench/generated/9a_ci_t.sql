SELECT * FROM cast_info AS ci, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND t.production_year BETWEEN 2005 AND 2015 AND ci.movie_id = t.id AND t.id = ci.movie_id;