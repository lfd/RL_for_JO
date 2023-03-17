SELECT * FROM title AS t, cast_info AS ci WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND t.production_year > 2000 AND t.id = ci.movie_id AND ci.movie_id = t.id;