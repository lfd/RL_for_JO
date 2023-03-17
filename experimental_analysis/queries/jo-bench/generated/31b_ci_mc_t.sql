SELECT * FROM movie_companies AS mc, title AS t, cast_info AS ci WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND mc.note LIKE '%(Blu-ray)%' AND t.production_year > 2000 AND (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%') AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;