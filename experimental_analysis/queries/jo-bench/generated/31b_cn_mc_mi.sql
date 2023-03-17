SELECT * FROM movie_info AS mi, company_name AS cn, movie_companies AS mc WHERE cn.name LIKE 'Lionsgate%' AND mc.note LIKE '%(Blu-ray)%' AND mi.info IN ('Horror', 'Thriller') AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;