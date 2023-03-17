SELECT * FROM movie_keyword AS mk, title AS t, company_name AS cn, movie_companies AS mc WHERE cn.name LIKE 'Lionsgate%' AND mc.note LIKE '%(Blu-ray)%' AND t.production_year > 2000 AND (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%') AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;