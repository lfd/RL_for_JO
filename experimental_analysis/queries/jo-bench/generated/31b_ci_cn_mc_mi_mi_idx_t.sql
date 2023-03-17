SELECT * FROM movie_companies AS mc, company_name AS cn, movie_info AS mi, movie_info_idx AS mi_idx, title AS t, cast_info AS ci WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND cn.name LIKE 'Lionsgate%' AND mc.note LIKE '%(Blu-ray)%' AND mi.info IN ('Horror', 'Thriller') AND t.production_year > 2000 AND (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%') AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mi_idx.movie_id AND mi_idx.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi_idx.movie_id AND mi_idx.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mi.movie_id = mi_idx.movie_id AND mi_idx.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi_idx.movie_id = mc.movie_id AND mc.movie_id = mi_idx.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;